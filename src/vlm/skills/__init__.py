"""Skill Registry — other 경로 영역별 Skill 구현.

PIPELINE.md §4 참조.
  - S2 PrintedTextReader
  - S3 HandwritingReader
  - S4 SealReader
  - S5 TableExtractor (2패스)
  - S6 SignatureDetector
  - S7 StructuredAggregator
"""

from src.vlm.skills.aggregator import StructuredAggregator
from src.vlm.skills.seal_reader import SealReader
from src.vlm.skills.signature_detector import SignatureDetector
from src.vlm.skills.table_extractor import TableExtractor

__all__ = [
    "SealReader",
    "SignatureDetector",
    "StructuredAggregator",
    "TableExtractor",
]
