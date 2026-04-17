"""파이프라인 오케스트레이터
군수 OCR 시스템 v2

P1~P6 전체 파이프라인을 8단계(고도화)로 조율합니다.

처리 경로 (주 경로):
  P1 → P2 → P2.5-A → P3-A → P2.5-B → P2.5-C → P3-B → P4 → P5 → P6
Fallback:
  P1 → P2 → P2.5-A → Fallback(T3~T5) → P4 → 검토 큐
완전 장애:
  원본 이미지 + 메타데이터 → 검토 큐

사용법:
    pipeline = PipelineOrchestrator(PipelineConfig(
        vllm_base_url="http://vllm-server:8000/v1",
    ))
    result = pipeline.run(doc_input)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.vlm.template_augmentor import TemplateAugmentorStats
    from src.vlm.skill_registry import SkillDispatchStats

from src.interfaces.enums import (
    OutputFormat,
    PipelineStatus,
    ProcessingPath,
    ReviewReason,
)
from src.interfaces.types import (
    DocumentInput,
    FieldValue,
    LayoutResult,
    PipelineOutput,
    PreprocessedImage,
    RawLayoutResult,
    ValidatedResult,
    VLMResult,
)
from src.vlm.logprobs_scorer import is_flagged as _is_flagged

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """파이프라인 설정."""

    # vLLM 서버
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_health_url: str = "http://localhost:8000/health"

    # 디바이스
    device: str = "gpu:0"

    # 모델 경로
    model_root: Optional[str] = None

    # 레이아웃 모델 선택 ("PP-DocLayout_plus-L" 또는 "PP-DocLayoutV3")
    layout_model_name: str = "PP-DocLayout_plus-L"

    # Layout 추론 서비스 URL (None이면 in-process, 값이면 HTTP 호출)
    layout_service_url: Optional[str] = None

    # 융합 모드: V3(구조) + plus-L(텍스트) 결과 합성 (layout_service_url 필요)
    layout_fusion_mode: bool = False

    # Fallback 설정
    fallback_enabled: bool = True
    fallback_base_url: str = "http://localhost:8081"

    # 검토 큐
    review_queue_enabled: bool = True
    review_queue_db_url: str = "sqlite:///data/review_queue.db"

    # DB 적재
    db_url: str = "sqlite:///data/ocr_results.db"

    # 출력 형식
    output_formats: list[OutputFormat] = None

    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = [OutputFormat.JSON, OutputFormat.XML, OutputFormat.CSV]


# ─────────────────────────────────────────────
#  실행 결과
# ─────────────────────────────────────────────

@dataclass
class PipelineResult:
    """파이프라인 실행 결과 — 각 단계의 출력을 보관."""

    doc_id: str
    status: PipelineStatus = PipelineStatus.SUCCESS
    processing_path: ProcessingPath = ProcessingPath.VLM

    # 단계별 출력
    p1_result: Optional[PreprocessedImage] = None
    p2_raw_result: Optional[RawLayoutResult] = None  # P2 원시
    p2_result: Optional[LayoutResult] = None          # P2.5-A 정제 후
    p3a_form_type: Optional[str] = None               # P3-A 서식 분류
    p3a_form_confidence: float = 0.0
    p3a_form_identifier: Optional[str] = None         # P3-A 서식 식별자 (예: "별지 제3-2호 서식")
    template_augmentor_stats: "Optional[TemplateAugmentorStats]" = None    # P2.5-A.5 TemplateAugmentorStats
    p2_5b_instructions: Optional[dict] = None         # P2.5-B InstructionRouter (region_id → InstructionSpec)
    p2_5c_groups: Optional[dict] = None               # P2.5-C ResolutionRouter (pixel_budget → [CroppedRegion])
    p3b_trace: list = field(default_factory=list)     # P3-B 영역별 vLLM 호출 trace
    p3_result: Optional[VLMResult] = None              # P3-B 추출 결과
    p4_result: Optional[ValidatedResult] = None
    output: Optional[PipelineOutput] = None

    # Skill Registry (other 경로)
    skill_results: list = field(default_factory=list)    # [SkillResult]
    skill_table_structures: list = field(default_factory=list)  # [TableStructure]
    skill_stats: "Optional[SkillDispatchStats]" = None                    # SkillDispatchStats

    # 실행 시간
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())


# ─────────────────────────────────────────────
#  PipelineOrchestrator
# ─────────────────────────────────────────────

class PipelineOrchestrator:
    """P1~P6 파이프라인 오케스트레이터.

    각 컴포넌트를 lazy 로드하여 실제 사용 시점에 초기화합니다.
    VLM 헬스 상태에 따라 주 경로/fallback/검토 큐를 자동 전환합니다.
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self._components: dict[str, Any] = {}

    # ═══════════════════════════════════════
    #  Lazy 컴포넌트 로드
    # ═══════════════════════════════════════

    def _get_p1(self):
        if "p1" not in self._components:
            from src.preprocess.preprocessor import P1Preprocessor
            self._components["p1"] = P1Preprocessor()
            logger.info("오케스트레이터: P1 초기화 완료")
        return self._components["p1"]

    def _get_p2(self):
        if "p2" not in self._components:
            from src.preprocess.layout_analyzer import P2LayoutAnalyzer, P2LayoutAnalyzerConfig
            config = P2LayoutAnalyzerConfig(
                device=self.cfg.device,
                model_name=self.cfg.layout_model_name,
                layout_service_url=self.cfg.layout_service_url,
                fusion_mode=self.cfg.layout_fusion_mode,
            )
            if self.cfg.model_root:
                config.model_dir = (
                    f"{self.cfg.model_root}/t2_layout/{self.cfg.layout_model_name}"
                )
            self._components["p2"] = P2LayoutAnalyzer(config)
            logger.info("오케스트레이터: P2 초기화 완료 (mode=%s, model=%s)",
                        self._components["p2"].mode.value,
                        self.cfg.layout_model_name)
        return self._components["p2"]

    def _get_p2_5a(self):
        """P2.5-A LayoutPostProcessor."""
        if "p2_5a" not in self._components:
            from src.preprocess.layout_postprocessor import LayoutPostProcessor
            self._components["p2_5a"] = LayoutPostProcessor()
            logger.info("오케스트레이터: P2.5-A 초기화 완료")
        return self._components["p2_5a"]

    def _get_p3a(self):
        """P3-A FormClassifier."""
        if "p3a" not in self._components:
            from src.vlm.form_classifier import FormClassifier, FormClassifierConfig
            config = FormClassifierConfig(
                vllm_base_url=self.cfg.vllm_base_url,
                monitor=self._get_health_monitor(),
            )
            self._components["p3a"] = FormClassifier(config)
            logger.info("오케스트레이터: P3-A 초기화 완료")
        return self._components["p3a"]

    def _get_p2_5b(self):
        """P2.5-B InstructionRouter."""
        if "p2_5b" not in self._components:
            from src.vlm.instruction_router import InstructionRouter
            self._components["p2_5b"] = InstructionRouter()
            logger.info("오케스트레이터: P2.5-B 초기화 완료")
        return self._components["p2_5b"]

    def _get_p2_5c(self):
        """P2.5-C ResolutionRouter."""
        if "p2_5c" not in self._components:
            from src.vlm.resolution_router import ResolutionRouter
            self._components["p2_5c"] = ResolutionRouter()
            logger.info("오케스트레이터: P2.5-C 초기화 완료")
        return self._components["p2_5c"]

    def _get_ocr_hint_provider(self):
        """OCR-augmented 힌트 제공자 (선택적, 저신뢰 재시도 보강)."""
        if "ocr_hint" not in self._components:
            from src.vlm.ocr_hint_provider import OCRHintProvider, OCRHintConfig
            cfg = OCRHintConfig()
            provider = OCRHintProvider(cfg)
            self._components["ocr_hint"] = provider
            logger.info(
                "오케스트레이터: OCRHintProvider 초기화 완료 (enabled=%s)",
                provider.enabled,
            )
        return self._components["ocr_hint"]

    def _get_p3b(self):
        """P3-B StructuredExtractor (OCRHintProvider 주입)."""
        if "p3b" not in self._components:
            from src.vlm.structured_extractor import (
                StructuredExtractor,
                StructuredExtractorConfig,
            )
            config = StructuredExtractorConfig(
                vllm_base_url=self.cfg.vllm_base_url,
                monitor=self._get_health_monitor(),
            )
            ocr_provider = self._get_ocr_hint_provider()
            self._components["p3b"] = StructuredExtractor(
                config=config,
                ocr_hint_provider=ocr_provider,
            )
            logger.info("오케스트레이터: P3-B 초기화 완료")
        return self._components["p3b"]

    def _get_template_augmentor(self):
        """P2.5-A.5 TemplateAugmentor — military 서식 템플릿 bbox 병합."""
        if "template_augmentor" not in self._components:
            from src.vlm.template_augmentor import TemplateAugmentor
            self._components["template_augmentor"] = TemplateAugmentor()
            logger.info("오케스트레이터: TemplateAugmentor 초기화 완료")
        return self._components["template_augmentor"]

    def _new_augmentor_stats(self, result: PipelineResult):
        """PipelineResult에 TemplateAugmentorStats를 부착하고 반환."""
        from src.vlm.template_augmentor import TemplateAugmentorStats
        stats = TemplateAugmentorStats()
        result.template_augmentor_stats = stats
        return stats

    def _get_skill_registry(self):
        """Skill Registry (other 경로)."""
        if "skill_registry" not in self._components:
            from src.vlm.skill_registry import SkillRegistry
            from src.vlm.vlm_client import VLMClient
            vlm = VLMClient(
                base_url=self.cfg.vllm_base_url,
                model_name="/models/gemma4/gemma-4-26b-a4b-it/",
                monitor=self._get_health_monitor(),
            )
            self._components["skill_registry"] = SkillRegistry(vlm_client=vlm)
            logger.info("오케스트레이터: SkillRegistry 초기화 완료")
        return self._components["skill_registry"]


    def _get_p4(self):
        if "p4" not in self._components:
            from src.postprocess.validator import P4Validator
            self._components["p4"] = P4Validator()
            logger.info("오케스트레이터: P4 초기화 완료")
        return self._components["p4"]

    def _get_p5(self):
        if "p5" not in self._components:
            from src.postprocess.serializer import P5Serializer, P5SerializerConfig
            config = P5SerializerConfig(formats=self.cfg.output_formats)
            self._components["p5"] = P5Serializer(config)
            logger.info("오케스트레이터: P5 초기화 완료")
        return self._components["p5"]

    def _get_p6(self):
        if "p6" not in self._components:
            from src.postprocess.db_loader import P6DBLoader, P6DBLoaderConfig
            config = P6DBLoaderConfig(db_url=self.cfg.db_url)
            self._components["p6"] = P6DBLoader(config)
            logger.info("오케스트레이터: P6 초기화 완료")
        return self._components["p6"]

    def _get_health_monitor(self):
        if "health" not in self._components:
            from src.pipeline.health_monitor import VLMHealthMonitor, HealthMonitorConfig
            config = HealthMonitorConfig(health_url=self.cfg.vllm_health_url)
            self._components["health"] = VLMHealthMonitor(config)
        return self._components["health"]

    def _get_fallback_policy(self):
        if "policy" not in self._components:
            from src.pipeline.fallback_policy import FallbackPolicy
            self._components["policy"] = FallbackPolicy()
        return self._components["policy"]

    def _get_fallback_service(self):
        if "fallback" not in self._components:
            # fallback_base_url이 설정되어 있으면 HTTP 클라이언트 사용 (Docker 분리 모드)
            # v1 모듈은 별도 컨테이너에서만 로드해야 함 (v2 enums 충돌 방지)
            if self.cfg.fallback_base_url:
                from src.fallback.fallback_http_client import FallbackHTTPClient
                self._components["fallback"] = FallbackHTTPClient(
                    base_url=self.cfg.fallback_base_url,
                )
                logger.info("오케스트레이터: Fallback HTTP 클라이언트 초기화 (url=%s)",
                            self.cfg.fallback_base_url)
            else:
                # fallback_base_url 미설정: fallback 비활성화 처리
                logger.warning(
                    "오케스트레이터: fallback_base_url 미설정 — "
                    "in-process fallback은 v1/v2 enums 충돌로 사용 불가. "
                    "fallback을 사용하려면 fallback 컨테이너를 기동하고 "
                    "fallback_base_url을 설정하세요."
                )
                # 헬스체크 시 항상 unhealthy로 응답하는 더미 반환
                class _DisabledFallback:
                    def is_healthy(self) -> bool:
                        return False
                    def process(self, *args, **kwargs):
                        return None
                self._components["fallback"] = _DisabledFallback()
        return self._components["fallback"]

    def _get_review_queue(self):
        if "review_queue" not in self._components:
            from src.postprocess.review_queue import ReviewQueue, ReviewQueueConfig
            config = ReviewQueueConfig(db_url=self.cfg.review_queue_db_url)
            self._components["review_queue"] = ReviewQueue(config)
        return self._components["review_queue"]

    # ═══════════════════════════════════════
    #  단계 실행 헬퍼
    # ═══════════════════════════════════════

    # ═══════════════════════════════════════
    #  Skill Registry 경로 (other 문서)
    # ═══════════════════════════════════════

    def _process_other_document(
        self,
        layout: LayoutResult,
        preprocessed: PreprocessedImage,
        doc_id: str,
        form_confidence: float,
        result: PipelineResult,
    ) -> VLMResult:
        """PIPELINE.md §5 other 경로 — S5 패스1 → 일반 태스크 + 셀 태스크 배치 디스패치."""
        from src.interfaces.enums import FormType, ProcessingPath
        from src.vlm.skill_registry import SkillDispatchStats

        registry = self._get_skill_registry()
        stats = SkillDispatchStats()

        # 1) 표 2패스 (S5)
        cell_results, structures = registry.dispatch_tables(
            layout=layout, preprocessed=preprocessed, stats=stats,
        )

        # 2) 비-표 영역 태스크
        non_table_tasks = registry.build_tasks(layout=layout, preprocessed=preprocessed)
        general_results = registry.dispatch(non_table_tasks, stats=stats)

        skill_results = general_results + cell_results
        result.skill_results = skill_results
        result.skill_table_structures = structures
        result.skill_stats = stats

        # 3) VLMResult 형태로 집계 (P4 이후 단계에 동일 인터페이스 제공)
        fields = _skill_results_to_fields(skill_results)
        overall = (
            sum(f.confidence for f in fields) / len(fields)
            if fields else 0.0
        )
        vlm_result = VLMResult(
            doc_id=doc_id,
            form_type=FormType.OTHER,
            form_confidence=form_confidence,
            schema_id="_general",
            fields=fields,
            tables=[],
            domain_codes=[],
            processing_time_ms=stats.total_ms,
            processing_path=ProcessingPath.SKILL_REGISTRY,
        )
        logger.info(
            "Skill Registry: %d 영역 처리 (표 %d, 일반 %d), avg_conf=%.3f, %.1fms",
            len(skill_results), len(cell_results), len(general_results),
            overall, stats.total_ms,
        )
        return vlm_result

    def _run_step(self, name: str, result: PipelineResult, fn) -> Any:
        """단일 단계 실행 + 시간 측정."""
        t0 = time.time()
        try:
            output = fn()
            elapsed = (time.time() - t0) * 1000
            result.timings[name] = round(elapsed, 1)
            logger.info("오케스트레이터: %s 완료 (%.1fms)", name, elapsed)

            if hasattr(output, "warnings") and output.warnings:
                # 동일 리스트 참조 시 무한 루프 방지: 스냅샷 복사 후 순회
                warnings_snapshot = list(output.warnings)
                for w in warnings_snapshot:
                    prefixed = f"[{name}] {w}"
                    if prefixed not in result.warnings:
                        result.warnings.append(prefixed)

            return output
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            result.timings[name] = round(elapsed, 1)
            result.errors.append(f"[{name}] {type(e).__name__}: {e}")
            logger.error("오케스트레이터: %s 실패 — %s", name, e)
            return None

    # ═══════════════════════════════════════
    #  메인 실행
    # ═══════════════════════════════════════

    def process(self, doc_input: DocumentInput):
        """단일 진입점 — 파일 포맷에 따라 분기.

        Returns:
            FileExt.PDF  → PdfDocumentResult (+ pages: list[PipelineResult])
            그 외        → PipelineResult
        """
        from src.interfaces.enums import FileExt
        if doc_input.file_ext == FileExt.PDF:
            return self.run_pdf(doc_input)
        return self.run(doc_input)

    def run_pdf(self, doc_input: DocumentInput):
        """PDF 전체를 페이지별로 처리하여 PdfDocumentResult 반환.

        각 페이지는 독립적으로 run()을 거치며, 한 페이지 실패가 다른 페이지에 전파되지 않음.
        `PipelineOutput`(직렬화용)과 `PipelineResult`(디버깅용)를 둘 다 반환.
        """
        import time as _t
        from src.interfaces.enums import FileExt, SourceType
        from src.interfaces.types import PdfDocumentResult
        from src.input.pdf_adapter import (
            PdfAdapter, PdfAdapterError, RENDER_DPI_DEFAULT,
        )

        start = _t.monotonic()
        adapter = PdfAdapter(render_dpi=doc_input.dpi_hint or RENDER_DPI_DEFAULT)

        try:
            pages = adapter.render(doc_input.doc_id, doc_input.raw_bytes)
        except PdfAdapterError as e:
            logger.error("[PDF] %s 렌더링 실패: %s", doc_input.doc_id, e)
            return PdfDocumentResult(
                doc_id=doc_input.doc_id,
                total_pages=0,
                pages=[],
                overall_status=PipelineStatus.FAILED,
                processing_ms=(_t.monotonic() - start) * 1000,
                warnings=[f"PdfAdapterError: {e}"],
            )

        page_outputs = []
        page_results = []  # PipelineResult 리스트 — 런타임 저장용 (PdfDocumentResult에 부착)
        for page in pages:
            page_doc_id = f"{doc_input.doc_id}_p{page.page_number:02d}"
            page_input = DocumentInput(
                doc_id=page_doc_id,
                raw_bytes=_ndarray_to_png_bytes(page.image_array),
                file_ext=FileExt.PNG,
                source_type=doc_input.source_type,
                dpi_hint=page.render_dpi,
                metadata={
                    **doc_input.metadata,
                    "parent_doc_id": doc_input.doc_id,
                    "page_number": page.page_number,
                    "total_pages": page.total_pages,
                    "pdf_source": True,
                },
            )
            page_result = self.run(page_input)
            page_results.append(page_result)
            if page_result.output is not None:
                page_outputs.append(page_result.output)

        doc_result = PdfDocumentResult(
            doc_id=doc_input.doc_id,
            total_pages=len(pages),
            pages=page_outputs,
            overall_status=_aggregate_pdf_status(page_outputs),
            processing_ms=(_t.monotonic() - start) * 1000,
            warnings=[w for p in pages for w in p.warnings],
            page_results=page_results,
        )

        logger.info(
            "[PDF] %s: %d 페이지 처리 완료 (status=%s, %.0fms)",
            doc_input.doc_id, len(pages), doc_result.overall_status.value,
            doc_result.processing_ms,
        )
        return doc_result

    def run(self, doc_input: DocumentInput) -> PipelineResult:
        """단일 이미지 파이프라인 실행.

        Args:
            doc_input: 원본 문서

        Returns:
            PipelineResult
        """
        result = PipelineResult(doc_id=doc_input.doc_id)
        logger.info("오케스트레이터: 파이프라인 시작 (doc_id=%s)", doc_input.doc_id)

        # ─── P1: 화질 보정 ───
        p1_out = self._run_step("P1", result, lambda: self._get_p1().process(doc_input))
        result.p1_result = p1_out
        if p1_out is None:
            result.status = PipelineStatus.FAILED
            return self._finalize(result)

        # ─── P2: 레이아웃 탐지 (원시) ───
        p2_raw = self._run_step("P2", result, lambda: self._get_p2().analyze(p1_out))
        result.p2_raw_result = p2_raw
        if p2_raw is None:
            result.status = PipelineStatus.FAILED
            return self._finalize(result)

        # ─── P2.5-A: LayoutPostProcessor (정제) ───
        p2_out = self._run_step("P2.5A", result, lambda: self._get_p2_5a().process(p2_raw))
        result.p2_result = p2_out
        if p2_out is None:
            result.status = PipelineStatus.FAILED
            return self._finalize(result)

        # ─── 경로 결정 (VLM / Fallback / 검토 큐) ───
        monitor = self._get_health_monitor()
        policy = self._get_fallback_policy()

        fallback_healthy = False
        if self.cfg.fallback_enabled:
            try:
                fallback_healthy = self._get_fallback_service().is_healthy()
            except Exception as e:
                result.warnings.append(f"[Fallback] 헬스체크 실패: {e}")

        path = policy.decide(
            vlm_healthy=monitor.is_healthy(),
            fallback_enabled=self.cfg.fallback_enabled,
            fallback_healthy=fallback_healthy,
        )

        # ─── VLM 주 경로: P3-A → P2.5-B → P2.5-C → P3-B ───
        p3_out: Optional[VLMResult] = None

        if path == "vlm":
            result.processing_path = ProcessingPath.VLM

            # P3-A: FormClassifier (서식 분류)
            classify_result = self._run_step(
                "P3A", result,
                lambda: self._get_p3a().classify(p1_out.image_array, result.warnings),
            )
            if classify_result is not None:
                # FormClassifier.classify() → (FormType, confidence, form_identifier)
                if len(classify_result) == 3:
                    form_type, form_confidence, form_identifier = classify_result
                else:
                    form_type, form_confidence = classify_result  # backward compat
                    form_identifier = None
                result.p3a_form_type = form_type.value if hasattr(form_type, "value") else str(form_type)
                result.p3a_form_confidence = form_confidence
                result.p3a_form_identifier = form_identifier

                # 스키마 로드 — form_type별 분기
                from src.domain.schema_registry import SchemaRegistry
                from src.interfaces.enums import FormType as _FT
                registry = SchemaRegistry()

                if form_type == _FT.OTHER:
                    # other 문서: Skill Registry 경로 — SKILL_ROUTING 기반 영역별 디스패치
                    result.processing_path = ProcessingPath.SKILL_REGISTRY
                    p3_out = self._run_step(
                        "SkillRegistry", result,
                        lambda: self._process_other_document(
                            layout=p2_out,
                            preprocessed=p1_out,
                            doc_id=doc_input.doc_id,
                            form_confidence=form_confidence,
                            result=result,
                        ),
                    )
                    if p3_out is not None:
                        result.p3_result = p3_out
                    # P2.5-B/C/P3-B 건너뜀 — P4~P6 계속
                    instructions = None
                    schema_id = "_general"
                    schema = None
                else:
                    if form_type == _FT.UNKNOWN:
                        # 군수 서식인데 유형 불명
                        schema_id = "_fallback"
                    else:
                        schema_id = result.p3a_form_type

                    schema = registry.load(schema_id)
                    if schema is None:
                        # 최종 fallback
                        schema = registry.load("_fallback")
                        schema_id = "_fallback"

                    # P2.5-A.5: TemplateAugmentor — 서식 템플릿으로 PP-DocLayout 누락 보완
                    aug_layout = self._run_step(
                        "P2.5A5", result,
                        lambda: self._get_template_augmentor().augment(
                            layout=p2_out,
                            form_type=form_type,
                            form_identifier=form_identifier,
                            stats=self._new_augmentor_stats(result),
                        ),
                    )
                    if aug_layout is not None:
                        p2_out = aug_layout
                        result.p2_result = aug_layout

                    # P2.5-B: InstructionRouter
                    instructions = self._run_step(
                        "P2.5B", result,
                        lambda: self._get_p2_5b().route_all(p2_out, form_type),
                    )
                    result.p2_5b_instructions = instructions

                    if instructions:
                        # P2.5-C: ResolutionRouter
                        groups = self._run_step(
                            "P2.5C", result,
                            lambda: self._get_p2_5c().route(p2_out, p1_out, instructions),
                        )
                        result.p2_5c_groups = groups

                        if groups:
                            # P3-B: StructuredExtractor (trace 수집)
                            p3_out = self._run_step(
                                "P3B", result,
                                lambda: self._get_p3b().extract(
                                    groups=groups,
                                    doc_id=doc_input.doc_id,
                                    form_type=form_type,
                                    form_confidence=form_confidence,
                                    schema_id=schema_id,
                                    schema=schema,
                                    warnings=result.warnings,
                                    trace=result.p3b_trace,
                                ),
                            )
                            result.p3_result = p3_out

            # VLM 실패 시 fallback 재시도
            if p3_out is None and self.cfg.fallback_enabled:
                result.warnings.append("[P3] VLM 파이프라인 실패 → fallback 전환")
                path = "fallback"

        if path == "fallback":
            result.processing_path = ProcessingPath.FALLBACK
            p3_out = self._run_step(
                "P3-Fallback", result,
                lambda: self._get_fallback_service().process(p1_out, p2_out),
            )
            result.p3_result = p3_out

            if p3_out is None:
                result.warnings.append("[Fallback] v1 T3~T5 실패 → 검토 큐 전환")
                path = "review_queue"

        if path == "review_queue" and p3_out is None:
            result.processing_path = ProcessingPath.NONE
            result.status = PipelineStatus.REVIEW
            if self.cfg.review_queue_enabled:
                self._enqueue_total_failure(result, doc_input)
            return self._finalize(result)

        if p3_out is None:
            result.status = PipelineStatus.FAILED
            return self._finalize(result)

        # ─── P4: 룰 검증 + 신뢰도 보정 ───
        p4_out = self._run_step(
            "P4", result,
            lambda: self._get_p4().validate(p3_out, result.processing_path),
        )
        result.p4_result = p4_out
        if p4_out is None:
            result.status = PipelineStatus.FAILED
            return self._finalize(result)

        # ─── 검토 큐 적재 (P4 실패 시) ───
        review_queue_id: Optional[str] = None
        if p4_out.review_required and self.cfg.review_queue_enabled:
            reason = ReviewReason.VALIDATION_FAILED
            if result.processing_path == ProcessingPath.FALLBACK:
                reason = ReviewReason.FALLBACK
            elif p4_out.flagged_fields:
                reason = ReviewReason.LOW_CONFIDENCE

            rq = self._get_review_queue()
            review_queue_id = rq.enqueue(p4_out, reason=reason)
            result.warnings.append(f"[검토큐] 적재 완료: {review_queue_id}")

        # ─── P5: 직렬화 (검토 필요 여부와 무관하게 항상 실행) ───
        json_out, xml_out, csv_rows = None, None, []
        serialized = self._run_step(
            "P5", result,
            lambda: self._get_p5().serialize(p4_out, form_type=p3_out.form_type),
        )
        if serialized:
            json_out, xml_out, csv_rows = serialized

        # ─── P6: DB 적재 ───
        from src.interfaces.enums import FormType as _FT
        status = PipelineStatus.SUCCESS
        if p3_out.form_type == _FT.OTHER:
            # other 문서: 범용 OCR 결과, 검토 큐 미적재
            status = PipelineStatus.OTHER_DOCUMENT
        elif p4_out.review_required:
            status = PipelineStatus.REVIEW
        elif result.processing_path == ProcessingPath.FALLBACK:
            status = PipelineStatus.PARTIAL

        output = self._run_step(
            "P6", result,
            lambda: self._get_p6().load(
                validated=p4_out,
                form_type=p3_out.form_type,
                json_output=json_out,
                xml_output=xml_out,
                csv_rows=csv_rows,
                status=status,
                processing_ms=result.total_ms,
                timings=result.timings,
                error_count=len(result.errors),
                warning_count=len(result.warnings),
                review_queue_id=review_queue_id,
            ),
        )
        result.output = output
        result.status = status

        return self._finalize(result)

    # ═══════════════════════════════════════
    #  완전 장애 처리
    # ═══════════════════════════════════════

    def _enqueue_total_failure(
        self,
        result: PipelineResult,
        doc_input: DocumentInput,
    ) -> None:
        """VLM + Fallback 모두 불가 시 원본을 검토 큐에 적재."""
        try:
            # 빈 ValidatedResult 생성
            empty_validated = ValidatedResult(
                doc_id=doc_input.doc_id,
                fields=[],
                tables=[],
                validation_errors=[],
                overall_confidence=0.0,
                review_required=True,
                flagged_fields=[],
                processing_path=ProcessingPath.NONE,
            )

            rq = self._get_review_queue()
            queue_id = rq.enqueue(
                empty_validated,
                reason=ReviewReason.TOTAL_FAILURE,
            )
            result.warnings.append(f"[검토큐] 완전 장애 적재: {queue_id}")
        except Exception as e:
            result.errors.append(f"[검토큐] 완전 장애 적재 실패: {e}")

    # ═══════════════════════════════════════
    #  최종 처리
    # ═══════════════════════════════════════

    def _finalize(self, result: PipelineResult) -> PipelineResult:
        """결과 마무리."""
        logger.info(
            "오케스트레이터: 파이프라인 완료 (doc_id=%s, status=%s, "
            "path=%s, total=%.0fms, errors=%d, warnings=%d)",
            result.doc_id, result.status.value,
            result.processing_path.value, result.total_ms,
            len(result.errors), len(result.warnings),
        )
        return result


# ─────────────────────────────────────────────
#  Skill Registry → VLMResult 변환 유틸
# ─────────────────────────────────────────────

def _skill_results_to_fields(skill_results: list) -> list[FieldValue]:
    """SkillResult 리스트 → FieldValue 리스트.

    region_id를 field_key로 사용. content는 raw/corrected 동일.
    data_type은 content_type 기반 추정.
    """
    fields: list[FieldValue] = []
    for sr in skill_results:
        data_type = "text"
        if sr.content_type == "signature":
            data_type = "flag"
        elif sr.content_type == "seal":
            data_type = "text"
        fields.append(
            FieldValue(
                field_key=sr.region_id,
                raw_value=sr.content,
                corrected_value=sr.content,
                data_type=data_type,
                confidence=sr.confidence,
                token_logprobs=[],
                is_flagged=_is_flagged(sr.confidence, data_type),
                region_id=sr.region_id,
            )
        )
    return fields


# ─────────────────────────────────────────────
#  PDF 어댑터 보조 유틸
# ─────────────────────────────────────────────

def _ndarray_to_png_bytes(image: "np.ndarray") -> bytes:
    """RGB numpy 배열 → PNG 바이트 (메모리 내 변환)."""
    import cv2
    import numpy as np  # noqa: F401
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")
    return buf.tobytes()


def _aggregate_pdf_status(page_outputs: list) -> PipelineStatus:
    """PDF 페이지별 상태 집계 → 전체 상태.

    - 모든 페이지 success → success
    - 모든 페이지 failed → failed
    - 그 외 (섞여 있거나 review/partial/other 포함) → partial
    """
    if not page_outputs:
        return PipelineStatus.FAILED
    statuses = {p.status for p in page_outputs}
    if statuses == {PipelineStatus.SUCCESS}:
        return PipelineStatus.SUCCESS
    if statuses == {PipelineStatus.FAILED}:
        return PipelineStatus.FAILED
    if statuses == {PipelineStatus.OTHER_DOCUMENT}:
        return PipelineStatus.OTHER_DOCUMENT
    return PipelineStatus.PARTIAL
