"""Wildlife analysis domain: detection, classification, and orchestration."""

from models.core import AnimalAnalyzer
from models.data_models import AnalysisResult, Classification, Detection

__all__ = [
    "AnimalAnalyzer",
    "AnalysisResult",
    "Classification",
    "Detection",
]
