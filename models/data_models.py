"""Shared wildlife analysis models and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from utils.types import BBox

# ------------------------------------------------------------------
# Result Models
# ------------------------------------------------------------------


@dataclass(slots=True)
class Detection:
    """
    Single detection output from MegaDetector.

    - bbox: pixel coordinates (x1, y1, x2, y2)
    - confidence: detector confidence in [0, 1]
    - label: raw MegaDetector category (``"animal"`` or ``"person"``)
    """

    bbox: BBox
    confidence: float
    label: str


@dataclass(slots=True)
class Classification:
    """
    Species classification for one detected animal.

    - species: predicted label (or ``"unknown"`` when below confidence threshold)
    - confidence: classifier confidence score in [0, 1]
    - bbox: detection bbox this classification corresponds to
    """

    species: str
    confidence: float
    bbox: BBox


@dataclass(slots=True)
class AnalysisResult:
    """
    Result of wildlife analysis for one image.

    - success=True: at least one animal classified successfully
    - success=False + detections set: animals detected but classification failed
    - success=False + detections empty: no animals detected or image load failed
    """

    detections: list[Detection]
    classifications: list[Classification]
    animal_count: int
    species_counts: dict[str, int] = field(default_factory=dict)
    success: bool = False
    error: str | None = None


# ------------------------------------------------------------------
# Model Protocols
# ------------------------------------------------------------------


class DetectorProtocol(Protocol):
    """Protocol for animal detector implementations."""

    def detect(self, image: Image.Image) -> list[Detection]:
        """Detect animals in one PIL image and return bounding boxes."""
        ...

    def detect_batch(self, images: list[Image.Image]) -> list[list[Detection]]:
        """
        Detect animals in multiple PIL images in input order.

        Implementations must return exactly ``len(images)`` lists,
        preserving input order.
        """
        ...


class ClassifierProtocol(Protocol):
    """Protocol for species classifier implementations."""

    def classify(self, crop: Image.Image) -> tuple[str, float]:
        """
        Classify one pre-cropped PIL image.

        Returns:
            ``(species, confidence)`` tuple.
        """
        ...

    def classify_batch(self, crops: list[Image.Image]) -> list[tuple[str, float]]:
        """
        Classify multiple pre-cropped PIL images in input order.

        Implementations must return exactly ``len(crops)`` tuples,
        preserving input order.
        """
        ...
