"""Transform domain: filename parsing and temporal deduplication."""

from pipeline.etl.transform.deduplicator import TemporalDeduplicator
from pipeline.etl.transform.filename_parser import (
    FilenameParser,
    PhototrapMetadata,
    detect_camera_type,
)

__all__ = [
    "FilenameParser",
    "PhototrapMetadata",
    "TemporalDeduplicator",
    "detect_camera_type",
]
