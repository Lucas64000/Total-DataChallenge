"""Public ETL API for camera-trap preprocessing."""

from pipeline.etl.class_catalog import ClassCatalog, load_class_catalog
from pipeline.etl.config import PathConfig, PreprocessingConfig
from pipeline.etl.etl_pipeline import ETLPipeline, PipelineResult
from pipeline.etl.extractor import Extractor
from pipeline.etl.timestamp_ocr import TimestampExtractor
from pipeline.etl.transform.deduplicator import TemporalDeduplicator

__all__ = [
    "ClassCatalog",
    "ETLPipeline",
    "PipelineResult",
    "Extractor",
    "PathConfig",
    "PreprocessingConfig",
    "TemporalDeduplicator",
    "TimestampExtractor",
    "load_class_catalog",
]
