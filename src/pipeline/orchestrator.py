"""파이프라인 오케스트레이터
군수 OCR 시스템 v2

P1~P6 전체 파이프라인을 6단계로 조율합니다.
v1(12단계)에서 재작성.

처리 경로:
  주 경로:     P1 → P2 → P3(VLM) → P4 → P5 → P6
  Fallback:    P1 → P2 → Fallback(T3~T5) → P4 → 검토 큐
  완전 장애:   원본 이미지 + 메타데이터 → 검토 큐

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
from typing import Any, Optional

from src.interfaces.enums import (
    OutputFormat,
    PipelineStatus,
    ProcessingPath,
    ReviewReason,
)
from src.interfaces.types import (
    DocumentInput,
    LayoutResult,
    PipelineOutput,
    PreprocessedImage,
    ValidatedResult,
    VLMResult,
)

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
    p2_result: Optional[LayoutResult] = None
    p3_result: Optional[VLMResult] = None
    p4_result: Optional[ValidatedResult] = None
    output: Optional[PipelineOutput] = None

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
            config = P2LayoutAnalyzerConfig(device=self.cfg.device)
            if self.cfg.model_root:
                config.model_dir = f"{self.cfg.model_root}/t2_layout/PP-DocLayout_plus-L"
            self._components["p2"] = P2LayoutAnalyzer(config)
            logger.info("오케스트레이터: P2 초기화 완료 (mode=%s)",
                        self._components["p2"].mode.value)
        return self._components["p2"]

    def _get_p3(self):
        if "p3" not in self._components:
            from src.vlm.gemma4_engine import Gemma4Engine, Gemma4EngineConfig
            config = Gemma4EngineConfig(vllm_base_url=self.cfg.vllm_base_url)
            self._components["p3"] = Gemma4Engine(config)
            logger.info("오케스트레이터: P3 초기화 완료")
        return self._components["p3"]

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
            from src.fallback.ocr_fallback_service import (
                OCRFallbackService,
                OCRFallbackConfig,
            )
            config = OCRFallbackConfig(device=self.cfg.device)
            self._components["fallback"] = OCRFallbackService(config)
            logger.info("오케스트레이터: Fallback 서비스 초기화 완료")
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

    def _run_step(self, name: str, result: PipelineResult, fn) -> Any:
        """단일 단계 실행 + 시간 측정."""
        t0 = time.time()
        try:
            output = fn()
            elapsed = (time.time() - t0) * 1000
            result.timings[name] = round(elapsed, 1)
            logger.info("오케스트레이터: %s 완료 (%.1fms)", name, elapsed)

            if hasattr(output, "warnings") and output.warnings:
                for w in output.warnings:
                    result.warnings.append(f"[{name}] {w}")

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

    def run(self, doc_input: DocumentInput) -> PipelineResult:
        """파이프라인 실행.

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

        # ─── P2: 구조 분석 ───
        p2_out = self._run_step("P2", result, lambda: self._get_p2().analyze(p1_out))
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

        # ─── P3: VLM 통합 추론 / Fallback ───
        p3_out: Optional[VLMResult] = None

        if path == "vlm":
            result.processing_path = ProcessingPath.VLM
            p3_out = self._run_step(
                "P3", result,
                lambda: self._get_p3().process(p1_out, p2_out),
            )
            result.p3_result = p3_out

            # VLM 실패 시 fallback 재시도
            if p3_out is None and self.cfg.fallback_enabled:
                result.warnings.append("[P3] VLM 실패 → fallback 전환")
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

        # ─── P5: 직렬화 ───
        json_out, xml_out, csv_rows = None, None, []
        if not p4_out.review_required or True:  # 검토 필요해도 직렬화는 실행
            serialized = self._run_step(
                "P5", result,
                lambda: self._get_p5().serialize(p4_out),
            )
            if serialized:
                json_out, xml_out, csv_rows = serialized

        # ─── P6: DB 적재 ───
        status = PipelineStatus.SUCCESS
        if p4_out.review_required:
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
            from src.postprocess.review_queue import ReviewQueue, ReviewQueueConfig
            from src.interfaces.types import ValidatedResult

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
