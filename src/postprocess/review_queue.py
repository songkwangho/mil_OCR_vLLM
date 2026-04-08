"""수동 검토 큐 — P4 검증 실패 문서 적재/관리
군수 OCR 시스템 v2

적재 조건 (P4에서 판정):
  - CRITICAL ValidationError → 긴급
  - LOW confidence 필드 존재 → 일반
  - Fallback 경로 처리 → 일반
  - VLM + Fallback 모두 불가 → 긴급

의존: sqlalchemy (SQLite)
"""

from __future__ import annotations

import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.interfaces.enums import (
    ProcessingPath,
    ReviewPriority,
    ReviewReason,
    ReviewStatus,
    Severity,
)
from src.interfaces.types import (
    ReviewQueueItem,
    ReviewQueueStats,
    ValidatedResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ReviewQueueConfig:
    """검토 큐 설정."""
    db_url: str = "sqlite:///data/review_queue.db"


# ─────────────────────────────────────────────
#  SQLAlchemy 모델
# ─────────────────────────────────────────────

_rq_engine = None
_rq_SessionLocal = None
_rq_Base = None


def _get_rq_db(db_url: str):
    """검토 큐 DB 초기화."""
    global _rq_engine, _rq_SessionLocal, _rq_Base

    if _rq_engine is not None:
        return _rq_engine, _rq_SessionLocal

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, declarative_base

    _rq_Base = declarative_base()
    _rq_engine = create_engine(db_url, echo=False)
    _rq_SessionLocal = sessionmaker(bind=_rq_engine)

    _define_rq_tables(_rq_Base)
    _rq_Base.metadata.create_all(_rq_engine)
    logger.info("ReviewQueue: DB 초기화 완료 (%s)", db_url)

    return _rq_engine, _rq_SessionLocal


def _define_rq_tables(Base):
    """검토 큐 테이블 정의."""
    from sqlalchemy import Column, String, Integer, Text, DateTime, Boolean

    class ReviewRecord(Base):
        __tablename__ = "review_queue"
        id = Column(Integer, primary_key=True, autoincrement=True)
        queue_id = Column(String(100), unique=True, nullable=False, index=True)
        doc_id = Column(String(100), nullable=False, index=True)
        enqueued_at = Column(DateTime, default=datetime.utcnow)
        priority = Column(String(20))
        reason = Column(String(50))
        processing_path = Column(String(20))
        validation_errors_json = Column(Text)
        flagged_fields_json = Column(Text)
        original_image_path = Column(Text)
        preprocessed_image_path = Column(Text)
        status = Column(String(20), default="pending")
        reviewer = Column(String(100))
        reviewed_at = Column(DateTime)
        corrected_fields_json = Column(Text)
        reviewer_notes = Column(Text)

    _define_rq_tables.ReviewRecord = ReviewRecord


class ReviewQueue:
    """수동 검토 큐 관리자.

    사용 예시:
        queue = ReviewQueue()
        queue_id = queue.enqueue(validated_result, reason=ReviewReason.VALIDATION_FAILED)
        stats = queue.get_stats()
    """

    def __init__(self, cfg: Optional[ReviewQueueConfig] = None):
        self.cfg = cfg or ReviewQueueConfig()
        self._available = False
        try:
            _get_rq_db(self.cfg.db_url)
            self._available = True
        except Exception as e:
            logger.warning("ReviewQueue: DB 초기화 실패: %s", e)

    def enqueue(
        self,
        validated: ValidatedResult,
        reason: ReviewReason,
        original_image_path: str = "",
        preprocessed_image_path: str = "",
    ) -> Optional[str]:
        """검토 큐에 문서 적재.

        Returns:
            queue_id (성공 시) 또는 None (실패 시)
        """
        import json

        queue_id = f"RQ-{datetime.now().strftime('%Y%m%d')}-{validated.doc_id}"

        # 우선순위 결정
        has_critical = any(
            e.severity == Severity.CRITICAL
            for e in validated.validation_errors
        )
        if has_critical or reason == ReviewReason.TOTAL_FAILURE:
            priority = ReviewPriority.CRITICAL
        else:
            priority = ReviewPriority.NORMAL

        if not self._available:
            logger.warning("ReviewQueue: DB 미사용 — queue_id=%s (메모리)", queue_id)
            return queue_id

        _, SessionLocal = _get_rq_db(self.cfg.db_url)
        ReviewRecord = _define_rq_tables.ReviewRecord

        session = SessionLocal()
        try:
            record = ReviewRecord(
                queue_id=queue_id,
                doc_id=validated.doc_id,
                priority=priority.value,
                reason=reason.value,
                processing_path=validated.processing_path.value,
                validation_errors_json=json.dumps(
                    [{"error_id": e.error_id, "type": e.error_type.value,
                      "severity": e.severity.value, "message": e.message}
                     for e in validated.validation_errors],
                    ensure_ascii=False,
                ),
                flagged_fields_json=json.dumps(validated.flagged_fields),
                original_image_path=original_image_path,
                preprocessed_image_path=preprocessed_image_path,
                status=ReviewStatus.PENDING.value,
            )
            session.add(record)
            session.commit()
            logger.info(
                "ReviewQueue: 적재 완료 (queue_id=%s, priority=%s, reason=%s)",
                queue_id, priority.value, reason.value,
            )
            return queue_id
        except Exception as e:
            session.rollback()
            logger.error("ReviewQueue: 적재 실패: %s", e)
            return None
        finally:
            session.close()

    def get_stats(self) -> ReviewQueueStats:
        """검토 큐 현황 통계."""
        if not self._available:
            return ReviewQueueStats(
                total_pending=0, total_in_review=0,
                critical_count=0, normal_count=0, avg_wait_minutes=0.0,
            )

        _, SessionLocal = _get_rq_db(self.cfg.db_url)
        ReviewRecord = _define_rq_tables.ReviewRecord

        session = SessionLocal()
        try:
            pending = session.query(ReviewRecord).filter_by(status="pending").count()
            in_review = session.query(ReviewRecord).filter_by(status="in_review").count()
            critical = session.query(ReviewRecord).filter_by(
                status="pending", priority="critical"
            ).count()
            normal = session.query(ReviewRecord).filter_by(
                status="pending", priority="normal"
            ).count()

            return ReviewQueueStats(
                total_pending=pending,
                total_in_review=in_review,
                critical_count=critical,
                normal_count=normal,
                avg_wait_minutes=0.0,  # TODO: 평균 대기 시간 산출
            )
        finally:
            session.close()
