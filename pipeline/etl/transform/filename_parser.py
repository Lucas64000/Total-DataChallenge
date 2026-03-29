"""
Parse camera-trap filenames into structured metadata.

The parser extracts the metadata required by the ML pipeline:
- location information used for location-based CV splits
- species label
- camera family used by OCR timestamp extraction

Date tokens from filenames are intentionally ignored because they may reflect
export/rename workflow dates instead of the real capture moment. Temporal logic
uses the timestamp extracted via OCR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.types import CameraType

# ------------------------------------------------------------------
# Camera Detection
# ------------------------------------------------------------------

def detect_camera_type(identifier: str) -> CameraType:
    """
    Detect camera type from filename or camera_id.

    Args:
        identifier: Filename or camera_id string

    Returns:
        'reconyx', 'boly', or 'unknown'
    """
    upper = identifier.upper()
    if "RCNX" in upper:
        return "reconyx"
    if "IMAG" in upper:
        return "boly"
    return "unknown"


# ------------------------------------------------------------------
# Metadata Model
# ------------------------------------------------------------------

@dataclass(slots=True)
class PhototrapMetadata:
    """Essential metadata extracted from a camera trap filename."""

    filename: str
    coord_n: str | None = None
    coord_w: str | None = None
    species: str | None = None
    camera_type: CameraType = "unknown"
    labeled: bool = False
    parse_success: bool = False
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a plain dict for DataFrame record construction.

        The returned keys match the columns expected by DataFrameBuilder._build_record.
        ``location_id`` is derived from coordinates and included explicitly because
        it is a computed property, not a raw field.
        """
        return {
            "filename": self.filename,
            "coord_n": self.coord_n,
            "coord_w": self.coord_w,
            "location_id": self.location_id,
            "species": self.species,
            "camera_type": self.camera_type,
            "labeled": self.labeled,
            "parse_success": self.parse_success,
            "parse_error": self.parse_error,
        }

    @property
    def location_id(self) -> str | None:
        """
        Unique location identifier for CV splits.

        Combines N and W coordinates.

        Returns:
            Canonical location identifier or ``None`` when coordinates are missing.
        """
        if self.coord_n and self.coord_w:
            # Some sources append a trailing letter (e.g. W1234a); drop it for stable grouping.
            coord_w_clean = re.sub(r"[a-z]$", "", self.coord_w)
            return f"{self.coord_n}_{coord_w_clean}"
        return None


# ------------------------------------------------------------------
# Filename Parser
# ------------------------------------------------------------------

class FilenameParser:
    """
    Parser for camera trap filenames.

    Expected stem format:
    {COUNTRY}_{COORD_N}_{COORD_W}_{DATE}_{SPECIES}_{CAMERA_ID}_{OPTIONAL_SUFFIX}

    Only fields required by the ML pipeline are extracted.
    """

    PATTERN = re.compile(
        r"^[A-Z]{2}_"  # Country ISO CODE
        r"(?P<coord_n>N[\d-]+)_"
        r"(?P<coord_w>W[\d-]+[a-z]?)_"
        r"\d{8}_"  # Date token (not captured - not the true capture date)
        r"(?P<species>[A-Za-z-]+)_"
        r"(?P<camera_id>[A-Za-z0-9-]+)"  # For camera_type detection
        r"(?:_.+)?$"  # Optional trailing suffix (e.g. _0001)
    )

    @classmethod
    def parse(cls, filename: str, labeled: bool = False) -> PhototrapMetadata:
        """
        Parse a filename and extract essential metadata.

        Args:
            filename: Filename (with or without extension)
            labeled: Whether this image has annotations

        Returns:
            PhototrapMetadata with extracted fields
        """
        name = Path(filename).stem
        metadata = PhototrapMetadata(filename=filename, labeled=labeled)

        match = cls.PATTERN.match(name)
        if not match:
            metadata.parse_error = "Filename does not match expected pattern"
            return metadata

        # Keep parsing strict to avoid silently producing wrong split metadata.
        groups = match.groupdict()
        metadata.coord_n = groups["coord_n"]
        metadata.coord_w = groups["coord_w"]
        metadata.species = groups["species"]
        metadata.camera_type = detect_camera_type(groups["camera_id"])
        metadata.parse_success = True
        return metadata
