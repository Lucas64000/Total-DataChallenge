"""
Define camera-specific OCR timestamp extraction profiles.

Each profile encapsulates three concerns:
- where to crop timestamp text in the image
- which timestamp regex patterns to try, in priority order
- how regex groups map to datetime components

Profiles are consumed by the timestamp extractor and parser so new camera types
can be added without changing extraction orchestration logic.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from utils.types import BBox, CameraType, TimestampGroupOrder


@dataclass(frozen=True, slots=True)
class CropRegion:
    """Crop region parameters as ratios of image dimensions."""

    x_start: float  # Horizontal start ratio (0.0 = left edge).
    y_start: float  # Vertical start ratio (0.0 = top edge).
    x_end: float    # Horizontal end ratio (1.0 = right edge).
    y_end: float    # Vertical end ratio (1.0 = bottom edge).

    def to_pixels(self, width: int, height: int) -> BBox:
        """
        Convert ratio-based region to pixel coordinates.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Bounding box tuple ``(x1, y1, x2, y2)`` in pixels.
        """
        return (
            int(width * self.x_start),
            int(height * self.y_start),
            int(width * self.x_end),
            int(height * self.y_end),
        )


@dataclass(frozen=True, slots=True)
class TimestampPattern:
    """
    A timestamp pattern with its regex and group ordering.

    group_order maps regex groups to (year, month, day, hour, minute, second).
    Example: (0, 1, 2, 3, 4, 5) for ISO format, (2, 1, 0, 3, 4, 5) for EU format.
    """

    pattern: re.Pattern[str]
    group_order: TimestampGroupOrder  # Regex group indexes for y, m, d, h, min, sec.


class CameraProfile(ABC):
    """
    Abstract base class for camera profiles.

    Each profile defines how to extract timestamps from a specific camera type.
    """

    @property
    @abstractmethod
    def camera_type(self) -> CameraType:
        """
        Return the camera type identifier.

        Returns:
            Camera type literal.
        """
        ...

    @property
    @abstractmethod
    def crop_region(self) -> CropRegion:
        """
        Return the crop region for timestamp location.

        Returns:
            Ratio-based crop region.
        """
        ...

    @property
    @abstractmethod
    def patterns(self) -> list[TimestampPattern]:
        """
        Return timestamp regex patterns in priority order.

        Returns:
            Ordered list of timestamp patterns.
        """
        ...


# Regex building blocks.
# Camera overlays may use ":" or "." between time fields (OCR can misread ":" as ".").
_STRICT_SEP = r"[:.]"

# Primary patterns keep explicit separators to avoid over-matching noisy OCR text.
_ISO_PATTERN = re.compile(
    rf"(\d{{4}})-(\d{{2}})-(\d{{2}})\s+(\d{{2}}){_STRICT_SEP}(\d{{2}}){_STRICT_SEP}(\d{{2}})"
)
_EU_PATTERN = re.compile(
    rf"(\d{{2}})-(\d{{2}})-(\d{{4}})\s+(\d{{2}}){_STRICT_SEP}(\d{{2}}){_STRICT_SEP}(\d{{2}})"
)

# Fallback patterns for cases where OCR drops separators in HH:MM:SS.
# `(?!\d)` avoids partial matches inside longer digit runs.
_ISO_NOSEP_PATTERN = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2})(\d{2})(\d{2})(?!\d)"
)
_EU_NOSEP_PATTERN = re.compile(
    r"(\d{2})-(\d{2})-(\d{4})\s+(\d{2})(\d{2})(\d{2})(?!\d)"
)

_ISO_ORDER: TimestampGroupOrder = (0, 1, 2, 3, 4, 5)  # y, m, d, h, min, sec
_EU_ORDER: TimestampGroupOrder = (2, 1, 0, 3, 4, 5)   # d, m, y in regex -> y, m, d in output


class ReconyxProfile(CameraProfile):
    """
    Profile for RECONYX camera traps.

    - Timestamp location: top-left corner
    - Format: ISO (YYYY-MM-DD HH:MM:SS)
    - Crop: (0, 0) to (40% width, 3% height)
    """

    @property
    def camera_type(self) -> CameraType:
        """
        Return profile camera type.

        Returns:
            ``"reconyx"``.
        """
        return "reconyx"

    @property
    def crop_region(self) -> CropRegion:
        """
        Return calibrated crop region for Reconyx timestamps.

        Returns:
            Ratio-based crop region.
        """
        # Tuned in notebooks/etl/03_ocr_calibration.ipynb for Reconyx overlays.
        return CropRegion(x_start=0.0, y_start=0.0, x_end=0.35, y_end=0.03)

    @property
    def patterns(self) -> list[TimestampPattern]:
        """
        Return timestamp parsing patterns for Reconyx.

        Returns:
            Ordered list of timestamp patterns.
        """
        return [
            TimestampPattern(_ISO_PATTERN, _ISO_ORDER),
            TimestampPattern(_ISO_NOSEP_PATTERN, _ISO_ORDER),
        ]


class BolyProfile(CameraProfile):
    """
    Profile for BOLY camera traps.

    - Timestamp location: bottom-right strip
    - Formats: BOTH ISO and EU observed in the wild
      - ISO: YYYY-MM-DD HH:MM:SS (e.g., 2016-05-28 09:39:54)
      - EU:  DD-MM-YYYY HH:MM:SS (e.g., 22-02-2016 19:20:39)
    - Crop: (35% width, 95% height) to (100%, 100%)
    """

    @property
    def camera_type(self) -> CameraType:
        """
        Return profile camera type.

        Returns:
            ``"boly"``.
        """
        return "boly"

    @property
    def crop_region(self) -> CropRegion:
        """
        Return calibrated crop region for Boly timestamps.

        Returns:
            Ratio-based crop region.
        """
        # Tuned in notebooks/etl/03_ocr_calibration.ipynb for Boly overlays.
        return CropRegion(x_start=0.45, y_start=0.96, x_end=1.0, y_end=1.0)

    @property
    def patterns(self) -> list[TimestampPattern]:
        """
        Return timestamp parsing patterns for Boly.

        Returns:
            Ordered list of timestamp patterns.
        """
        # Order matters: ISO first (most frequent), then EU fallbacks.
        return [
            TimestampPattern(_ISO_PATTERN, _ISO_ORDER),      # ISO with explicit separators.
            TimestampPattern(_ISO_NOSEP_PATTERN, _ISO_ORDER),
            TimestampPattern(_EU_PATTERN, _EU_ORDER),        # EU with explicit separators.
            TimestampPattern(_EU_NOSEP_PATTERN, _EU_ORDER),
        ]


class UnknownProfile(CameraProfile):
    """
    Fallback profile for unknown camera types.

    - Timestamp location: full top strip (wider search area)
    - Tries both ISO and EU formats
    - Recall-first fallback when camera detection returns ``unknown``
    """

    @property
    def camera_type(self) -> CameraType:
        """
        Return profile camera type.

        Returns:
            ``"unknown"``.
        """
        return "unknown"

    @property
    def crop_region(self) -> CropRegion:
        """
        Return fallback crop region for unknown camera types.

        Returns:
            Ratio-based crop region.
        """
        # Recall-first fallback: search the whole top strip when camera is unknown.
        return CropRegion(x_start=0.0, y_start=0.0, x_end=1.0, y_end=0.10)

    @property
    def patterns(self) -> list[TimestampPattern]:
        """
        Return fallback timestamp parsing patterns.

        Returns:
            Ordered list of timestamp patterns.
        """
        # Keep strict patterns first, then no-separator fallbacks.
        return [
            TimestampPattern(_ISO_PATTERN, _ISO_ORDER),
            TimestampPattern(_ISO_NOSEP_PATTERN, _ISO_ORDER),
            TimestampPattern(_EU_PATTERN, _EU_ORDER),
            TimestampPattern(_EU_NOSEP_PATTERN, _EU_ORDER),
        ]


# Singleton profile registry reused for every extraction call.
_PROFILES: dict[CameraType, CameraProfile] = {
    "reconyx": ReconyxProfile(),
    "boly": BolyProfile(),
    "unknown": UnknownProfile(),
}


def get_profile(camera_type: CameraType) -> CameraProfile:
    """
    Get the profile implementation for a camera type.

    Args:
        camera_type: Camera type literal.

    Returns:
        Matching camera profile object.
    """
    return _PROFILES[camera_type]
