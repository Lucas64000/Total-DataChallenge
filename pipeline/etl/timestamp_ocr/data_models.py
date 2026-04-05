"""Shared timestamp extraction models and protocols."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from PIL import Image

from utils.types import CameraType

# ------------------------------------------------------------------
# Result Models
# ------------------------------------------------------------------

@dataclass(slots=True)
class TimestampResult:
    """
    Result of timestamp extraction.

    - success=True: timestamp is set and OCR text is available
    - success=False + raw_text set: OCR ran but parsing failed
    - success=False + raw_text None: extraction failed before OCR completed
    """

    timestamp: datetime | None
    camera_type: CameraType
    raw_text: str | None = None
    success: bool = False
    error: str | None = None


# ------------------------------------------------------------------
# OCR Protocols
# ------------------------------------------------------------------

class OCREngineProtocol(Protocol):
    """Protocol for OCR engine implementations."""

    def read(self, image: Image.Image) -> str:
        """Read text from one PIL image."""
        ...

    def read_batch(self, images: list[Image.Image]) -> list[str]:
        """Read text from multiple PIL images in input order.

        Implementations must return exactly ``len(images)`` strings,
        preserving input order.
        """
        ...
