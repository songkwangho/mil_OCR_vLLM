"""P6 — DB 적재 모듈
군수 OCR 시스템 v2 · Stage 3

v1 T12에서 이관 + v2 스키마 조정:
  - ExtractedDocument → ValidatedResult 입력 변경
  - processing_path 컬럼 추가
  - review_queue_id 컬럼 추가
  - source_line_id 제거 (v2에서 line_id 없음)

DB 스키마:
  documents     — 문서 메타 (doc_id, form_type, status, confidence, processing_path)
  fields        — 추출 필드 (doc_id FK, field_key, raw_value, corrected_value)
  pipeline_runs — 파이프라인 실행 기록

의존: sqlalchemy
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.interfaces.enums import FormType, OutputFormat, PipelineStatus, ProcessingPath
from src.interfaces.types import PipelineOutput, ValidatedResult

logger = logging.getLogger(__name__)


@dataclass
class P6DBLoaderConfig:
    """P6 DB 적재 설정."""
    db_url: str = "sqlite:///data/ocr_results.db"
    store_serialized: bool = True
    fallback_dir: str = "data/db_fallback"


# ─────────────────────────────────────────────
#  SQLAlchemy 모델 정의
# ─────────────────────────────────────────────

_engine = None
_SessionLocal = None
_Base = None


def _get_db(db_url: str):
    """DB 엔진 및 세션 초기화 (지연 로딩)."""
    global _engine, _SessionLocal, _Base

    if _engine is not None:
        return _engine, _SessionLocal

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, declarative_base

    _Base = declarative_base()
    _engine = create_engine(db_url, echo=False)
    _SessionLocal = sessionmaker(bind=_engine)

    _define_tables(_Base)
    _Base.metadata.create_all(_engine)
    logger.info("P6: DB 초기화 완료 (%s)", db_url)

    return _engine, _SessionLocal


def _define_tables(Base):
    """SQLAlchemy ORM 테이블 정의 (v2 스키마)."""
    from sqlalchemy import Column, String, Float, Boolean, Integer, Text, DateTime, ForeignKey

    class DocumentRecord(Base):
        __tablename__ = "documents"
        id = Column(Integer, primary_key=True, autoincrement=True)
        doc_id = Column(String(100), unique=True, nullable=False, index=True)
        form_type = Column(String(50))
        status = Column(String(20))
        processing_path = Column(String(20))
        confidence = Column(Float)
        review_required = Column(Boolean, default=False)
        review_queue_id = Column(String(100))
        json_output = Column(Text)
        xml_output = Column(Text)
        processed_at = Column(DateTime)
        created_at = Column(DateTime, default=datetime.utcnow)

    class FieldRecord(Base):
        __tablename__ = "fields"
        id = Column(Integer, primary_key=True, autoincrement=True)
        doc_id = Column(String(100), ForeignKey("documents.doc_id"), nullable=False, index=True)
        field_key = Column(String(100))
        raw_value = Column(Text)
        corrected_value = Column(Text)
        data_type = Column(String(30))
        confidence = Column(Float)
        is_flagged = Column(Boolean, default=False)

    class PipelineRunRecord(Base):
        __tablename__ = "pipeline_runs"
        id = Column(Integer, primary_key=True, autoincrement=True)
        doc_id = Column(String(100), ForeignKey("documents.doc_id"), nullable=False, index=True)
        status = Column(String(20))
        processing_path = Column(String(20))
        processing_ms = Column(Float)
        timings_json = Column(Text)
        error_count = Column(Integer, default=0)
        warning_count = Column(Integer, default=0)
        created_at = Column(DateTime, default=datetime.utcnow)

    _define_tables.DocumentRecord = DocumentRecord
    _define_tables.FieldRecord = FieldRecord
    _define_tables.PipelineRunRecord = PipelineRunRecord


class P6DBLoader:
    """P6 DB 적재기.

    사용법:
        loader = P6DBLoader()
        output = loader.load(
            validated=validated_result,
            json_output=json_str,
            status=PipelineStatus.SUCCESS,
            processing_ms=1234.5,
        )
    """

    def __init__(self, cfg: Optional[P6DBLoaderConfig] = None):
        self.cfg = cfg or P6DBLoaderConfig()
        self._db_available = False

        try:
            _get_db(self.cfg.db_url)
            self._db_available = True
        except Exception as e:
            logger.warning("P6: DB 초기화 실패 → JSON fallback 모드 (%s)", e)

    def load(
        self,
        validated: ValidatedResult,
        form_type: Optional[FormType] = None,
        json_output: Optional[str] = None,
        xml_output: Optional[str] = None,
        csv_rows: Optional[list[dict[str, str]]] = None,
        status: PipelineStatus = PipelineStatus.SUCCESS,
        processing_ms: float = 0.0,
        timings: Optional[dict[str, float]] = None,
        error_count: int = 0,
        warning_count: int = 0,
        review_queue_id: Optional[str] = None,
    ) -> PipelineOutput:
        """ValidatedResult를 DB에 적재하고 PipelineOutput을 반환."""
        db_record_ids: list[str] = []

        if self._db_available:
            try:
                record_ids = self._insert_to_db(
                    validated, form_type, json_output, xml_output,
                    status, processing_ms, timings,
                    error_count, warning_count, review_queue_id,
                )
                db_record_ids = record_ids
            except Exception as e:
                logger.error("P6: DB 적재 실패 → fallback: %s", e)
                self._save_fallback(validated, json_output, xml_output)
        else:
            self._save_fallback(validated, json_output, xml_output)

        logger.info(
            "P6: 적재 완료 (doc_id=%s, records=%d, review=%s)",
            validated.doc_id, len(db_record_ids), review_queue_id or "N",
        )

        return PipelineOutput(
            doc_id=validated.doc_id,
            status=status,
            processing_path=validated.processing_path,
            form_type=form_type,
            json_output=json_output,
            xml_output=xml_output,
            csv_rows=csv_rows or [],
            db_record_ids=db_record_ids,
            review_queue_id=review_queue_id,
            processing_ms=processing_ms,
        )

    def _insert_to_db(
        self,
        validated: ValidatedResult,
        form_type: Optional[FormType],
        json_output: Optional[str],
        xml_output: Optional[str],
        status: PipelineStatus,
        processing_ms: float,
        timings: Optional[dict[str, float]],
        error_count: int,
        warning_count: int,
        review_queue_id: Optional[str],
    ) -> list[str]:
        """DB 삽입."""
        _, SessionLocal = _get_db(self.cfg.db_url)
        DocumentRecord = _define_tables.DocumentRecord
        FieldRecord = _define_tables.FieldRecord
        PipelineRunRecord = _define_tables.PipelineRunRecord

        session = SessionLocal()
        record_ids = []
        try:
            existing = session.query(DocumentRecord).filter_by(doc_id=validated.doc_id).first()
            if existing:
                existing.form_type = form_type.value if form_type else None
                existing.status = status.value
                existing.processing_path = validated.processing_path.value
                existing.confidence = validated.overall_confidence
                existing.review_required = validated.review_required
                existing.review_queue_id = review_queue_id
                existing.json_output = json_output if self.cfg.store_serialized else None
                existing.xml_output = xml_output if self.cfg.store_serialized else None
                existing.processed_at = datetime.utcnow()
                session.query(FieldRecord).filter_by(doc_id=validated.doc_id).delete()
            else:
                doc_record = DocumentRecord(
                    doc_id=validated.doc_id,
                    form_type=form_type.value if form_type else None,
                    status=status.value,
                    processing_path=validated.processing_path.value,
                    confidence=validated.overall_confidence,
                    review_required=validated.review_required,
                    review_queue_id=review_queue_id,
                    json_output=json_output if self.cfg.store_serialized else None,
                    xml_output=xml_output if self.cfg.store_serialized else None,
                    processed_at=datetime.utcnow(),
                )
                session.add(doc_record)
            session.flush()
            record_ids.append(f"doc:{validated.doc_id}")

            for f in validated.fields:
                field_record = FieldRecord(
                    doc_id=validated.doc_id,
                    field_key=f.field_key,
                    raw_value=f.raw_value,
                    corrected_value=f.corrected_value,
                    data_type=f.data_type,
                    confidence=f.confidence,
                    is_flagged=f.is_flagged,
                )
                session.add(field_record)
            record_ids.append(f"fields:{len(validated.fields)}")

            run_record = PipelineRunRecord(
                doc_id=validated.doc_id,
                status=status.value,
                processing_path=validated.processing_path.value,
                processing_ms=processing_ms,
                timings_json=json.dumps(timings or {}, default=str),
                error_count=error_count,
                warning_count=warning_count,
            )
            session.add(run_record)
            record_ids.append(f"run:{validated.doc_id}")

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return record_ids

    def _save_fallback(
        self,
        validated: ValidatedResult,
        json_output: Optional[str],
        xml_output: Optional[str],
    ) -> None:
        """DB 적재 실패 시 로컬 JSON 파일로 저장."""
        fallback_dir = Path(self.cfg.fallback_dir)
        fallback_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = fallback_dir / f"{validated.doc_id}_{ts}.json"

        data = {
            "doc_id": validated.doc_id,
            "overall_confidence": validated.overall_confidence,
            "review_required": validated.review_required,
            "fields": [
                {
                    "field_key": f.field_key,
                    "corrected_value": f.corrected_value,
                    "confidence": f.confidence,
                    "is_flagged": f.is_flagged,
                }
                for f in validated.fields
            ],
            "json_output": json_output,
            "fallback_reason": "db_unavailable",
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2, default=str)

        logger.info("P6: fallback 저장 → %s", path)
